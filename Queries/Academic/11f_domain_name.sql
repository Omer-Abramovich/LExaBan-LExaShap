SELECT
    domain.name
FROM
    conference, domain_conference, domain, domain_keyword, keyword
WHERE
    conference.cid = domain_conference.cid
    AND domain_conference.did = domain.did
    AND domain.did = domain_keyword.did
    AND domain_keyword.kid = keyword.kid
    AND keyword.keywod = 'Natural Language Processing'
GROUP BY domain.name